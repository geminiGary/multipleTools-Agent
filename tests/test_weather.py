"""天气工具单元测试。"""
from tools.weather import WeatherTool


def test_known_city():
    assert "22°C" in WeatherTool().run(city="北京")


def test_unknown_city_has_fallback():
    result = WeatherTool().run(city="火星")
    assert "火星" in result
    assert "示例数据" in result


def test_schema_shape():
    schema = WeatherTool().openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert "city" in schema["function"]["parameters"]["properties"]
