/*
MIT License

Copyright (c) 2018 Eric Laukien

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <vector>
#include <random>
#include <unordered_set>
#include <memory>

using namespace std;

class tsetlin_machine {
public:
    struct clause {
        vector<int> automata_states;
        unordered_set<int> inclusions;
        int state;

        void inclusion_update(int k);
        void apply_feedback(int k, int feedback, int num_states);
        void modifyI(const vector<int>& inputs, int num_states, bernoulli_distribution& one_by_s, mt19937 &rng);
        void modifyII(const vector<int>& inputs, int num_states);
    };

    vector<clause> clauses;
    int num_states;

    uniform_real_distribution<double> dist01;
    bernoulli_distribution one_by_s;

    tsetlin_machine(int num_inputs, int num_clauses, int num_states, double s, mt19937& rng);

    int predict(const vector<int> &inputs);
    void learn(const vector<int> &inputs, int target, int T, mt19937 &rng);
};


class multiclass_tsetlin_machine {
    vector<unique_ptr<tsetlin_machine>> class_clauses;
    vector<int> class_votes;
    uniform_int_distribution<> other_class;

public:
    multiclass_tsetlin_machine(int num_inputs, int num_classes, int num_clauses, int num_states, double s, mt19937& rng);

    int predict(const vector<int> &inputs);
    void predict_by_class(const vector<int> &inputs, vector<int>& class_votes);
    void learn(const vector<int> &inputs, int target_class, int T, mt19937 &rng);
};
