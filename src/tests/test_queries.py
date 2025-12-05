import unittest

from sea.query.operators import ANDNOTOperator, ANDOperator, OROperator, PhraseOperator, TermOperator
from sea.query.parser import QueryParser
from sea.query.specs import Operators
from sea.utils.config import Config

CFG = Config()

class TestQueryEngine(unittest.TestCase):
    def test_parse_OR_query(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("a OR b")
        self.assertIsInstance(root_operator, OROperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)

    def test_parse_AND_query(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("a AND a ANDNOT b")
        self.assertIsInstance(root_operator, ANDNOTOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], ANDOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], TermOperator)
        self.assertEqual(len(root_operator.children[0].children), 2)

    def test_parse_PHRASE_query(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("'hello world' AND 'foo bar'")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], PhraseOperator)
        self.assertIsInstance(root_operator.children[1], PhraseOperator)
        self.assertEqual(root_operator.children[0].phrase, "'hello world'")
        self.assertEqual(root_operator.children[1].phrase, "'foo bar'")

        
    def test_parse_bracket_query(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("(cat OR dog) and tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertTrue(isinstance(root_operator.children[0], OROperator) or isinstance(root_operator.children[1], OROperator))
        self.assertTrue(isinstance(root_operator.children[0], TermOperator) or isinstance(root_operator.children[1], TermOperator))
        if isinstance(root_operator.children[0], OROperator):
            or_operator = root_operator.children[0]
        else:
            or_operator = root_operator.children[1]
        self.assertIsInstance(or_operator.children[0], TermOperator)
        self.assertIsInstance(or_operator.children[1], TermOperator)
        self.assertEqual(or_operator.children[0].phrase, "cat")
        self.assertEqual(or_operator.children[1].phrase, "dog")

    def test_parse_nested_bracket_query(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("(cat andnot (blue or green) or dog) and tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertIsInstance(root_operator.children[0], OROperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], ANDNOTOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0].children[1], OROperator)

    def test_parse_PHRASE_single_tick(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("it's a test")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertEqual(len(root_operator.children), 3)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertListEqual([child.phrase for child in root_operator.children], ["it's", "a", "test"])

    def test_operator_identification(self):
        operator_class = Operators.get_EnumOperator("AND")
        self.assertEqual(operator_class, Operators.AND) 
        operator_class = Operators.get_EnumOperator("or")
        self.assertEqual(operator_class, Operators.OR)
        operator_class = Operators.get_EnumOperator("andnot")
        self.assertEqual(operator_class, Operators.ANDNOT)
        operator_class = Operators.get_EnumOperator("c.a.t")
        self.assertEqual(operator_class, Operators.TERM)

    def test_single_phrase(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("'hello'")
        self.assertIsInstance(root_operator, TermOperator)
        self.assertEqual(root_operator.phrase, "'hello'")

    def test_spaces(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("cat ")
        self.assertIsInstance(root_operator, TermOperator)
        self.assertEqual(root_operator.phrase, "cat")

    def test_multiple_terms(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("cat dog tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertEqual(len(root_operator.children), 3)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[2], TermOperator)

    def test_multiple_terms_with_and(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("cat AND dog tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], ANDOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)

    def test_multiple_term_with_or(self):
        engine = QueryParser(CFG)
        root_operator = engine.process_phrase2query("wish or expect or your or consider or charity or goal or will or purchase or albert or relationship or want or you or what or do")
        self.assertIsInstance(root_operator, OROperator)

        root_operator.execute(None, None)




if __name__ == '__main__':
    unittest.main()
