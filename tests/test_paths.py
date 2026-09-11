from sift_agent.paths import unique_dir


def test_unique_dir_creates_base(tmp_path):
    d = unique_dir(tmp_path / "run")
    assert d.is_dir()
    assert d.name == "run"


def test_unique_dir_suffixes_on_collision(tmp_path):
    first = unique_dir(tmp_path / "run")
    second = unique_dir(tmp_path / "run")
    third = unique_dir(tmp_path / "run")
    assert first != second != third
    assert second.name == "run-2"
    assert third.name == "run-3"


def test_unique_dir_creates_parents(tmp_path):
    d = unique_dir(tmp_path / "deep" / "nested" / "run")
    assert d.is_dir()


def test_unique_dir_thread_race(tmp_path):
    import threading

    base = tmp_path / "race"
    reserved = []
    lock = threading.Lock()

    def go():
        d = unique_dir(base)
        with lock:
            reserved.append(d)

    threads = [threading.Thread(target=go) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reserved) == 16
    assert len(set(reserved)) == 16
    assert all(p.is_dir() for p in reserved)
