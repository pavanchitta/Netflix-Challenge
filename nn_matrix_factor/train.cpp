#include <vector>
#include <stdio.h>

#include "nn_matrix_factor.h"

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
    string train_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/train.dta";
    string test_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/probe.dta";
    int K = 40;
    double eta = 0.01;
    double reg = 1;
    NNSVD m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename, 3.6095161972728063);
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/nn_matrix_factor.txt", preds);

    return 0;
}
