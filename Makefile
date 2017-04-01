.PHONY: check, docs

check:
	py.test --cov=fcdiff ${TEST_ARGS} test_fcdiff

docs:
	cd doc && $(MAKE) html

