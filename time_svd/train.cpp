#include <vector>
#include <stdio.h>

#include "time_svd.h"

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
    string all_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/all.dta";
    string train_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/train.dta";
    string test_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/probe.dta";
    int K = 1000;
    TimeSVD m(458293, 17770, K, all_filename, train_filename, test_filename, valid_filename);
    cout << "Starting training" << endl;
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/svd_test_preds_time_svd2.txt", preds);
    return 0;
}
