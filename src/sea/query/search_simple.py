import sys
import time
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from sea.ltr.serve_tfr import TFRReranker
from sea.query.splade import SpladeEncoder
from sea.utils.config_wrapper import Config
from omegaconf import DictConfig


def search_documents(
    *,
    cfg: DictConfig,
    query: str,
    splade_encoder: Optional[SpladeEncoder],
    max_output_result: int,
):
    if splade_encoder is not None:
        query = " ".join(splade_encoder.expand(query))

    rr = TFRReranker.load(model_path=cfg.LTR.SERVE_MODEL_PATH)
    return rr.rerank(query, candidate_topn=cfg.LTR.CANDIDATE_TOPN, topk=max_output_result), query
    
    
    

def main():
    cfg = Config(load=True)
    history = InMemoryHistory()
    session = PromptSession(history=history)
    max_output_result = cfg.SEARCH.MAX_RESULTS if cfg.SEARCH.MAX_RESULTS is not None else 10
    splade_encoder = None
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

        t0 = time.time()    
        documents, final_query = search_documents(
            cfg=cfg,
            query=query,
            splade_encoder=splade_encoder,
            max_output_result=max_output_result,
        )
        elapsed = (time.time() - t0)*1000
        history.append_string(final_query)

        if documents:
            for doc in documents:
                doc.pprint(verbose=cfg.SEARCH.VERBOSE_OUTPUT, loud=True)

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
