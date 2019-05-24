#include <vector>
#include <stdio.h>

#include "asymm_svd_plustime.h"

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

    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/train2.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/probe.dta";
    string all_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/all.dta";
    int K = 1;
    double eta = 0.01;
    double reg = 0.02;
    Model m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename, all_filename);
    cout << "Starting training with " << K << " factors" << endl;
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/pavanchitta/CS156b-Netflix/svd_asymm2.txt", preds);

    return 0;
}
