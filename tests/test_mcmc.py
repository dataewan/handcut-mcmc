import handcut_mcmc


def test_should_accept(monkeypatch):
    def always_1():
        return 1

    monkeypatch.setattr("handcut_mcmc.mcmc.np.random.rand", always_1)
    # We should accept if the product of the propsal likelihood and proposal
    # prior is greater than the same calculated for the current point
    assert handcut_mcmc.mcmc.should_accept(1, 20, 1, 1) is True
    assert handcut_mcmc.mcmc.should_accept(1, 1, 1, 20) is True


def test_should_reject(monkeypatch):
    def always_1():
        return 1

    monkeypatch.setattr("handcut_mcmc.mcmc.np.random.rand", always_1)

    # Check that we reject things correctly
    assert handcut_mcmc.mcmc.should_accept(20, 1, 1, 1) is False
    assert handcut_mcmc.mcmc.should_accept(1, 1, 20, 1) is False

    # Whenever we have equal products we should by convention reject
    assert handcut_mcmc.mcmc.should_accept(1, 1, 1, 1) is False
