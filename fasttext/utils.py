def get_trian_data(windows: list, window_size, ingnore="<DUMMY>"):
    train_data = []
    for window in windows:
        for i in range(window_size * 2 + 1):
            if window[i] in ingnore or window[window_size] in ingnore:
                continue  # min_count
            if i == window_size or window[i] == '<DUMMY>':
                continue
            train_data.append((window[window_size], window[i]))
