#include <vector>
#include <stdio.h>

#include "svd_plustime2.h"

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

    string train_filename = "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/train_probe.dta";
    string test_filename =  "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/probe.dta";
    int K = 100;
    double eta = 0.01;
    double reg = 0.02;
    Model m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename);
    cout << "Starting training with " << K << " factors" << endl;
    m.train();

    vector<double> preds_qual = m.predict();
    vector<double> preds_train = m.predict_train();

    writeToFile("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/KNNoutputs/qual_preds.txt", preds_qual);
    writeToFile("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/KNNoutputs/train_preds.txt", preds_train);

    return 0;
}
