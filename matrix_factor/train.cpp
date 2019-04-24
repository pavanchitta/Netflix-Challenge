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
    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/all.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";

    Model m(458293, 17770, 20, 0.01, 0.1, train_filename, test_filename, 3.512599);
    m.train();
    vector<double> preds = m.predict();
    writeToFile("/Users/pavanchitta/CS156b-Netflix/svd_test_preds.txt", preds);

    return 0;
}
