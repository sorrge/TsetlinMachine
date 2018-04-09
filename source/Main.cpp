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

#include "tsetlin/TsetlinMachine.h"

#include <iostream>
#include <ctime>
#include <algorithm>

using namespace std;

//#include "simple_demos.h"
#include "noisy_xor_task.h"


int main() {
//    wavy_demo();
//    xor_demo();
    int trials = 100;
    vector<double> accuracies(trials, 0);
    int finished = 0;
    double accuracy_sum;
#pragma omp parallel for
    for(int r = 0; r < trials; ++r) {
        noisy_xor nx(r);
        accuracies[r] = nx.test();
        ++finished;
        accuracy_sum += accuracies[r];
        cout << accuracies[r] << "\t" << accuracy_sum / finished << endl;
    }

    cout << "average accuracy: " << accumulate(accuracies.begin(), accuracies.end(), 0.0) / trials << endl;

    return 0;
}

