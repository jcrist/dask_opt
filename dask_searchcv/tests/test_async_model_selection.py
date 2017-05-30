from sklearn.datasets import load_digits


def test_criterion():
    criterion = Criterion()


def test_async_searchcv():
    digits = load_digits()

    model = SVC(kernel='rbf')

    # ended up tuning these parameters a bit :-P
    param_space = {'C': stats.expon(0.00001, 20),
                   'gamma': stats.expon(0.00001, 0.5),
                   'class_weight': [None, 'balanced']}

    n_splits = 3
    n_iter = 100000
    random_state = 1
    client = Client()

    search = AsyncRandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        cv=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        client=client
    )

    X = digits.data
    y = digits.target
    fit_params = {}

    start_t = time.time()
    search.fit_async(digits.data, digits.target)

    client.shutdown()

    print("Search finished")
    print("best_score {}; best_params - {}".format(search.best_score_,
                                                   search.best_params_))
    print("Async fit took {:.3f} seconds".format(time.time()-start_t))

