#pragma once

#include <random>
#include <algorithm>
#include "tsetlin/TsetlinMachine.h"

using namespace std;


class noisy_xor {
    const int dim = 12, clauses = 20, epochs = 200, num_samples = 10000, T = 15, automata_states = 200;
    const double prob_output_flip = 0.4, s = 3.9;

    mt19937 rng;
    bernoulli_distribution input, flip_output;

    vector<vector<int>> train_input, train_output, test_input, test_output;
    vector<int> train_indices;
    int real_input_idx_1, real_input_idx_2;

    void gen_data(int num_samples_to_generate, vector<vector<int>>& inputs, vector<vector<int>>& outputs) {
        inputs.resize(num_samples_to_generate, vector<int>(dim));
        outputs.resize(num_samples_to_generate, vector<int>(1));

        for(int i = 0; i < num_samples_to_generate; ++i) {
            for (int j = 0; j < dim; ++j)
                inputs[i][j] = input(rng);

            outputs[i][0] = inputs[i][real_input_idx_1] ^ inputs[i][real_input_idx_2];
        }
    }

public:
    noisy_xor(int seed) : rng(seed), flip_output(prob_output_flip) {
        reset();
    }

    void reset() {
        uniform_int_distribution<> idx1(0, dim - 1), idx2(1, dim - 1);
        real_input_idx_1 = idx1(rng);
        real_input_idx_2 = (real_input_idx_1 + idx2(rng)) % dim;

        gen_data(num_samples / 2, train_input, train_output);
        gen_data(num_samples / 2, test_input, test_output);
        train_indices.resize(train_output.size());
        for(int i = 0; i < (int)train_output.size(); ++i) {
            if (flip_output(rng))
                train_output[i][0] ^= 1;

            train_indices[i] = i;
        }
    }

    double test() {
        //tsetlin_machine tm(dim, clauses, automata_states, s, rng);
        multiclass_tsetlin_machine tm(dim, 2, clauses, automata_states, s, rng);

        for (int e = 0; e < epochs; e++) {
            shuffle(train_indices.begin(), train_indices.end(), rng);

            for (int i = 0; i < (int)train_input.size(); ++i) {
                tm.learn(train_input[train_indices[i]], train_output[train_indices[i]][0], T, rng);
            }
        }

        int num_correct = 0;
        for (int i = 0; i < (int)test_input.size(); ++i) {
            int out = tm.predict(test_input[i]);
            num_correct += out == test_output[i][0];
        }

        return (double)num_correct / test_input.size();
    }
};
