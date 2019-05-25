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

    string train_filename = "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/train.dta";
    string test_filename = "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/probe.dta";
    int K = 350;
    Model m(458293, 17770, K, train_filename, test_filename, valid_filename);
    cout << "Starting training" << endl;
    m.train();
    vector<double> preds = m.predict();
    vector<double> preds_valid = m.predict_valid();

    writeToFile("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/outputs/svd++350_qual.txt", preds);
    writeToFile("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/outputs/svd++350_valid.txt", preds_valid);

    return 0;
}
