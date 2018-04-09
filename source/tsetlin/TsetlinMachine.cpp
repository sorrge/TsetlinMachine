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

#include "TsetlinMachine.h"
#include <iostream>
#include <algorithm>

tsetlin_machine::tsetlin_machine(int num_inputs, int num_clauses, int _num_states, double s, mt19937& rng) : num_states(_num_states), one_by_s(1 / s) {
    num_states = _num_states;
    uniform_int_distribution<> state_init(0, 1);

    clauses.resize(num_clauses);

    for (int ci = 0; ci < num_clauses; ci++) {
        clauses[ci].automata_states.resize(num_inputs * 2, 0);
        for (int j = 0; j < num_inputs * 2; ++j) {
            clauses[ci].automata_states[j] = state_init(rng);
            clauses[ci].inclusion_update(j);
        }
    }
}

int tsetlin_machine::predict(const vector<int> &inputs) {
    int sum = 0;

    for (int j = 0; j < (int)clauses.size(); j++) {
        int state = 1;

        for (unordered_set<int>::const_iterator cit = clauses[j].inclusions.begin(); cit != clauses[j].inclusions.end(); cit++) {
            int ai = *cit;
            int neg = ai % 2;
            state &= (inputs[ai / 2] ^ neg);
        }

        clauses[j].state = state;
        sum += (j % 2 == 0 ? state : -state);
    }

    return sum;
}

void tsetlin_machine::clause::apply_feedback(int k, int feedback, int num_states) {
    int& the_state = automata_states[k];
    int state_change = feedback * (the_state > 0 ? 1 : -1);
    if (state_change > 0 && the_state < num_states / 2 || state_change < 0 && the_state > -num_states / 2 + 1) {
        int prev_state = the_state;
        the_state += state_change;
        if ((prev_state > 0) != (the_state > 0))
            inclusion_update(k);
    }
}

void tsetlin_machine::clause::inclusion_update(int k) {
    int inclusion = automata_states[k] > 0;

    unordered_set<int>::iterator it = inclusions.find(k);

    if (inclusion) {
        if (it == inclusions.end())
            inclusions.insert(k);
    }
    else {
        if (it != inclusions.end())
            inclusions.erase(it);
    }
}

void tsetlin_machine::clause::modifyI(const vector<int>& inputs, int num_states, bernoulli_distribution& one_by_s, mt19937 &rng) {
    for (int ai = 0; ai < (int)automata_states.size(); ai++) {
        int neg_literal = ai % 2;
        int truth_value_of_target_literal = inputs[ai / 2] ^ neg_literal;
        int inclusion = automata_states[ai] > 0;

        if (state) {
            if (truth_value_of_target_literal) {
                if (inclusion) {
                    if (!one_by_s(rng))
                        apply_feedback(ai, 1, num_states);
                }
                else {
                    if (!one_by_s(rng))
                        apply_feedback(ai, -1, num_states);
                }
            }
            else {
                if (inclusion) {
                    // NA
                    cerr << "Something impossible happened" << endl;
                }
                else {
                    if (one_by_s(rng))
                        apply_feedback(ai, 1, num_states);
                }
            }
        }
        else {
            if (truth_value_of_target_literal) {
                if (inclusion) {
                    if (one_by_s(rng))
                        apply_feedback(ai, -1, num_states);
                }
                else {
                    if (one_by_s(rng))
                        apply_feedback(ai, 1, num_states);
                }
            }
            else {
                if (inclusion) {
                    if (one_by_s(rng))
                        apply_feedback(ai, -1, num_states);
                }
                else {
                    if (one_by_s(rng))
                        apply_feedback(ai, 1, num_states);
                }
            }
        }
    }
}

void tsetlin_machine::clause::modifyII(const vector<int>& inputs, int num_states) {
    for (int ai = 0; ai < automata_states.size(); ai++) {
        int neg_literal = ai % 2;
        int input = inputs[ai / 2] ^ neg_literal;
        int inclusion = automata_states[ai] > 0;

        if (state) {
            if (!input) {
                if (!inclusion)
                    apply_feedback(ai, -1, num_states);
            }
        }
    }
}

void tsetlin_machine::learn(const vector<int>& inputs, int target, int T, mt19937 &rng) {
    int sum = predict(inputs);
    int clamped_sum = min(T, max(-T, sum));
    double rescale = 1.0 / (2 * T);

    double probFeedBack0 = (T - clamped_sum) * rescale;
    double probFeedBack1 = (T + clamped_sum) * rescale;

    for (int j = 0; j < (int)clauses.size(); j += 2) {                              // 1.7
        if (target) {                                                               // 1.8
            if (dist01(rng) <= probFeedBack0)                                       // 1.9
                clauses[j].modifyI(inputs, num_states, one_by_s, rng);// 1.10
        }
        else {                                                                      // 1.12
            if (dist01(rng) <= probFeedBack1)                                       // 1.13
                clauses[j].modifyII(inputs, num_states);                            // 1.14
        }                                                                           // 1.16
    }                                                                               // 1.17

    for (int j = 1; j < (int)clauses.size(); j += 2) {
        if (target) {
            if (dist01(rng) <= probFeedBack0)
                clauses[j].modifyII(inputs, num_states);
        }
        else {
            if (dist01(rng) <= probFeedBack1)
                clauses[j].modifyI(inputs, num_states, one_by_s, rng);
        }
    }
}


multiclass_tsetlin_machine::multiclass_tsetlin_machine(int num_inputs, int num_classes, int num_clauses, int num_states, double s, mt19937 &rng) : other_class(1, num_classes - 1) {
    int clauses_per_class = num_clauses / num_classes;
    for(int i = 0; i < num_classes; ++i)
        class_clauses.push_back(make_unique<tsetlin_machine>(num_inputs, clauses_per_class, num_states, s, rng));

    class_votes.resize(num_classes);
}


void multiclass_tsetlin_machine::predict_by_class(const vector<int> &inputs, vector<int>& class_votes) {
    for(int c = 0; c < (int)class_clauses.size(); ++c)
        class_votes[c] = class_clauses[c]->predict(inputs);
}


int multiclass_tsetlin_machine::predict(const vector<int> &inputs) {
    predict_by_class(inputs, class_votes);
    return (int)distance(class_votes.begin(), max_element(class_votes.begin(), class_votes.end()));
}


void multiclass_tsetlin_machine::learn(const vector<int> &inputs, int target_class, int T, mt19937 &rng) {
    predict_by_class(inputs, class_votes);
    int negative_target_class = (target_class + other_class(rng)) % class_clauses.size();
    double rescale = 1.0 / (2 * T);

    for (int j = 0; j < (int)class_clauses[target_class]->clauses.size(); ++j) {
        int clamped_sum = min(T, max(-T, class_votes[target_class]));
        double probFeedBack0 = (T - clamped_sum) * rescale;
        if (class_clauses[target_class]->dist01(rng) <= probFeedBack0)
            if (j % 2 == 0)
                class_clauses[target_class]->clauses[j].modifyI(inputs, class_clauses[target_class]->num_states, class_clauses[target_class]->one_by_s, rng);
            else
                class_clauses[target_class]->clauses[j].modifyII(inputs, class_clauses[target_class]->num_states);
    }

    for (int j = 0; j < (int)class_clauses[negative_target_class]->clauses.size(); ++j) {
        int clamped_sum = min(T, max(-T, class_votes[negative_target_class]));
        double probFeedBack1 = (T + clamped_sum) * rescale;
        if (class_clauses[negative_target_class]->dist01(rng) <= probFeedBack1)
            if (j % 2 == 0)
                class_clauses[negative_target_class]->clauses[j].modifyII(inputs, class_clauses[negative_target_class]->num_states);
             else
                class_clauses[negative_target_class]->clauses[j].modifyI(inputs, class_clauses[negative_target_class]->num_states, class_clauses[negative_target_class]->one_by_s, rng);
    }
}
