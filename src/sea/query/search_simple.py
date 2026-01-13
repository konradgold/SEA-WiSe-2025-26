import sys
import time
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.query.splade import SpladeEncoder
from sea.ranking.io_wrapper import bm25
from sea.utils.chunker import Chunker
from sea.utils.config_wrapper import Config


def search_documents(
    *,
    retriever,
    query: str,
    splade_encoder: Optional[SpladeEncoder],
    tokenizer: TokenizerAbstract
):
    if splade_encoder is not None:
        query = " ".join(splade_encoder.expand(query))

    query_listed = tokenizer.tokenize(query)
    time_start = time.perf_counter()
    return retriever(query_listed), query, time_start
    
    
    

def main():
    cfg = Config(load=True)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    ranker = bm25()
    splade_encoder = None
    chunker = Chunker(cfg=cfg)
    tokenizer = get_tokenizer(cfg)
    if cfg.SEARCH.EXPAND_QUERIES:
        from sea.query.splade import SpladeEncoder
        splade_encoder = SpladeEncoder(cfg=cfg)

    while True:
        try:
            query = session.prompt("\nEnter your search query (or 'quit' to exit):\n> ")
        except EOFError:
            break  # Exit on Ctrl-D

        if query.lower() == 'quit':
            break
        if not query:
            continue   

        documents, final_query, t0 = search_documents(
            retriever=ranker,
            query=query,
            splade_encoder=splade_encoder,
            tokenizer=tokenizer
        )
        elapsed = (time.perf_counter() - t0)*1000
        history.append_string(query)
        chunker.set_query(final_query.split())

        if documents:
            for doc in documents:
                doc.pprint(verbose=cfg.SEARCH.VERBOSE_OUTPUT, loud=True, chunker=chunker)

            print(f"\nFound {len(documents)} matches in {elapsed:.2f} milliseconds.")
        else:
            print("\nNo matches found.")
        print(f"Query: {final_query}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
