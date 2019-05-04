
#include <vector>
#include <stdio.h>

#include "matrix_factor_bias.h"

using namespace std;


void gridSearch() {

    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/train.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/probe.dta";

    int best_K = -1;
    int best_eta = -1;
    int best_reg = -1;

    int min_err = 10000; // Arbitrary upper limit

    for (int K = 30; K < 60; K += 10) {
        for (double eta = 0.05; eta < 0.2; eta += 0.05) {
            for (double reg = 0.05; reg < 0.3; reg += 0.05) {
                Model m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename, 3.512599);
                cout << " Training model with parameters (K, eta, reg) "
                    << K << " " << eta << " " << reg << endl;
                m.train();
                double probe_err = m.validErr();
                if (probe_err < min_err) {
                    best_K = m.params.K;
                    best_eta = m.params.eta;
                    best_reg = m.params.reg;
                    min_err = probe_err;
                }
            }
        }
    }

    cout << "Best probe error is " << min_err << " with parameters (K, eta, reg) "
        << best_K << " " << best_eta << " " << best_reg << endl;
}

int main() {
    girdSearch();
    return 0;
}
