#include <vector>
#include <stdio.h>

#include "svd_plus.h"

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

    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/train.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/probe.dta";
    int K = 40;
    Model m(458293, 17770, K, train_filename, test_filename, valid_filename);
    cout << "Starting training" << endl;
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/svd_test_preds.txt", preds);

    return 0;
}