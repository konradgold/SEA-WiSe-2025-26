import unittest
from queries import Operators, QueryEngine, OROperator, ANDOperator, ANDNOTOperator, PhraseOperator, TermOperator


class TestQueryEngine(unittest.TestCase):
    def test_process_OR_query(self):
        engine = QueryEngine()
        root_operator = engine._process_phrase2query("a OR b")
        self.assertIsInstance(root_operator, OROperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)

    def test_process_AND_query(self):
        engine = QueryEngine()
        root_operator = engine._process_phrase2query("a AND b AND not c")
        self.assertIsInstance(root_operator, ANDNOTOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], ANDOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)

    def test_operator_identification(self):
        operator_class = Operators.get_operator_for_identifier("AND")
        self.assertEqual(operator_class, Operators.AND.value["operator"])
        operator_class = Operators.get_operator_for_identifier("OR")
        self.assertEqual(operator_class, Operators.OR.value["operator"])
        operator_class = Operators.get_operator_for_identifier("andnot")
        self.assertEqual(operator_class, Operators.ANDNOT.value["operator"])
        operator_class = Operators.get_operator_for_identifier("c.a.t")
        self.assertEqual(operator_class, Operators.TERM.value["operator"])


if __name__ == '__main__':
    unittest.main()