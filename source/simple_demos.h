#pragma once


void xor_demo() {
    mt19937 rng(time(nullptr));

    vector<vector<int>> inputs = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    vector<vector<int>> outputs = {
        { 0, 1 },
        { 1, 0 },
        { 1, 0 },
        { 0, 1 },
    };

    TsetlinMachine tm;

    tm.create(2, 2, 10, 10, rng);

    float avgErr = 1.0f;

    for (int e = 0; e < 1000; e++) {
        vector<int> in = inputs[e % 4];

        vector<int> out = tm.activate(in);

        bool correct = false;

        if (in[0] == in[1]) {
            if (out[0] == 0 && out[1] == 1)
                correct = true;
        }
        else {
            if (out[0] == 1 && out[1] == 0)
                correct = true;
        }

        avgErr = 0.99f * avgErr + 0.01f * !correct;

        cout << in[0] << " " << in[1] << " -> " << out[0] << " " << out[1] << " | " << avgErr << endl;

        tm.learn(outputs[e % 4], 4.0f, 4, rng);
    }
}


void wavy_demo() {
    mt19937 rng(time(nullptr));

    int res = 10;
    int mem = 8;

    TsetlinMachine tm;

    tm.create(res * mem, res, 100, 10, rng);

    vector<vector<int>> memStates(mem, vector<int>(res, 0));

    float avgErr = 1.0f;

    vector<int> out(res, 0);

    for (int e = 0; e < 5000; e++) {
        // Generate input
        int index = min(res - 1, max(0, static_cast<int>((sin(e * 0.234f) * 0.5f + 0.5f) * res)));

        vector<int> state(res, 0);
        state[index] = 1;

        if (e != 0)
            tm.learn(state, 5.0f, 10, rng);

        memStates.insert(memStates.begin(), state);
        memStates.pop_back();

        // Flatten
        vector<int> flat(res * mem);

        for (int m = 0; m < mem; m++)
            for (int r = 0; r < res; r++) {
                flat[r + m * res] = memStates[m][r];
            }

        out = tm.activate(flat);

        for (int i = 0; i < out.size(); i++)
            cout << (out[i] ? "⬛" : ".") << " ";

        cout << endl;
    }

    // Recall
    for (int e = 0; e < 100; e++) {
        memStates.insert(memStates.begin(), out);
        memStates.pop_back();

        // Flatten
        vector<int> flat(res * mem);

        for (int m = 0; m < mem; m++)
            for (int r = 0; r < res; r++) {
                flat[r + m * res] = memStates[m][r];
            }

        out = tm.activate(flat);

        int index = 0;

        for (; index < res; index++)
            if (out[index])
                break;

        if (index == res)
            index = 0;

        cout << (index / static_cast<float>(res - 1)) << endl;
    }
}

