from tools.calculator import CalculatorTool


def test_basic_add():
    assert CalculatorTool().run(expression="2+3") == "2+3 = 5"


def test_precedence():
    assert "= 14" in CalculatorTool().run(expression="2+3*4")


def test_invalid_expression():
    assert "无法计算" in CalculatorTool().run(expression="2+")
