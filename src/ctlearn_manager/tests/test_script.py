def test_fib(capsys, monkeypatch):
    from template.cli.fib import main

    monkeypatch.setattr("sys.argv", ["fib", "10"])
    main()

    captured = capsys.readouterr()
    assert captured.out == "55\n"
