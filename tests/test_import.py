def test_polpy_imports() -> None:
    try:
        from polpy import poldata, polresponse, polarizationlike
    except ImportError:
        print("Polpy package not imported successfully")