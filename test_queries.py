import unittest
from query_parser import QueryParser
from query_operators import ANDNOTOperator, ANDOperator, OROperator, PhraseOperator, TermOperator
from query_operator_specs import Operators


class TestQueryEngine(unittest.TestCase):
    def test_parse_OR_query(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("a OR b")
        self.assertIsInstance(root_operator, OROperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)

    def test_parse_AND_query(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("a AND a ANDNOT b")
        self.assertIsInstance(root_operator, ANDNOTOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], ANDOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)

    def test_parse_PHRASE_query(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("'hello world' AND 'foo bar'")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], PhraseOperator)
        self.assertIsInstance(root_operator.children[1], PhraseOperator)
        self.assertEqual(root_operator.children[0].phrase, "'hello world'")
        self.assertEqual(root_operator.children[1].phrase, "'foo bar'")

        
    def test_parse_bracket_query(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("(cat OR dog) and tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertIsInstance(root_operator.children[0], OROperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)
        self.assertEqual(root_operator.children[0].children[0].phrase, "cat")
        self.assertEqual(root_operator.children[0].children[1].phrase, "dog")

    def test_parse_nested_bracket_query(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("(cat andnot (blue or green) or dog) and tree")
        self.assertIsInstance(root_operator, ANDOperator)
        self.assertIsInstance(root_operator.children[0], OROperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], ANDNOTOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0].children[1], OROperator)



    def test_parse_PHRASE_single_tick(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("it's a test")
        self.assertIsInstance(root_operator, TermOperator)
        self.assertEqual(root_operator.phrase, "it's")

    def test_operator_identification(self):
        operator_class = Operators.get_EnumOperator("AND")
        self.assertEqual(operator_class, Operators.AND) 
        operator_class = Operators.get_EnumOperator("or")
        self.assertEqual(operator_class, Operators.OR)
        operator_class = Operators.get_EnumOperator("andnot")
        self.assertEqual(operator_class, Operators.ANDNOT)
        operator_class = Operators.get_EnumOperator("c.a.t")
        self.assertEqual(operator_class, Operators.TERM)




    def test_DEMO_for_execution(self):
        engine = QueryParser()
        root_operator = engine.process_phrase2query("banana AND banana OR cherry ANDNOT banana")
        result = root_operator.execute()
        expected_ids = {"ID-banana", "ID-cherry"}
        self.assertEqual(result, expected_ids)


if __name__ == '__main__':
    unittest.main()
