#include <vector>
#include <stdio.h>

#include "matrix_factor_bias.h"

using namespace std;

void writeToFile(string filename, vector<double> preds) {
    ofstream out;
    out.open(filename);

    for (auto val: preds) {
        out << val << "\n";
    }
    out.close();
}

int main() {
    //string filename = "/Users/vigneshv/code/CS156b-Netflix/matrix_factor/mu/all.dta";
    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/train_probe.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/probe.dta";
    int K = 40;
    double eta = 0.01;
    double reg = 0.01;
    SVD m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename, 3.6095161972728063);
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/pavanchitta/CS156b-Netflix/svd_only_test_preds_1.txt", preds);

    return 0;
}