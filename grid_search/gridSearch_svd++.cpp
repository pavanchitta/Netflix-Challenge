#include <vector>
#include <stdio.h>

//#include "../matrix_factor_bias/matrix_factor_bias.h"
//#include "../svd++only/svd_plus.h"
#include "../svd++/svd_plustime2.h"
//#include "../time_svd/time_svd.h"

using namespace std;

void writeToFile(string filename, vector<double> preds) {
    ofstream out;
    out.open(filename);

    for (auto val: preds) {
        out << val << "\n";
    }
    out.close();
}

void gridSearch() {
    string train_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/train2.dta";
    string test_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/qual.dta";
    string valid_filename = "/Users/pavanchitta/CS156b-Netflix/data/um/probe.dta";

    // for (int K = 50; K < 1000; K += 100) {
    //     for (double eta = 0.01; eta < 0.2; eta += 0.05) {
    //         for (double reg = 0.005; reg < 0.03; reg += 0.005) {
    //             SVD m(458293, 17770, K, eta, reg, train_filename, test_filename, valid_filename, 3.6095161972728063);
    //             cout << " Training model with parameters (K, eta, reg) "
    //                << K << " " << eta << " " << reg << endl;
    //             m.train();
    //
    //             double probe_err = m.validErr();
    //             stringstream filename_stream;
    //             filename_stream << "basic_svd_" << "factors-" << K << "-eta-" << eta <<
    //                 "-reg-" << reg << "-err-" << probe_err;
    //             string s = filename_stream.str();
    //             cout << "Writing to: " << s << endl;
    //
    //             vector<double> preds = m.predict();
    //             writeToFile(s, preds);
    //         }
    //     }
    // }


//     for (double init = 2.0; init < 4.0; init += 0.3) {
//         for (int epochs = 30; epochs < 55; epochs++) {
//             for (int K = 30; K < 60; K += 10) {
//                 TimeSVD m(458293, 17770, K, train_filename, test_filename, valid_filename, epochs, pow(10, init));
//                 m.train();
//
//                 double probe_err = m.validErr();
//                 stringstream filename_stream;
//                 filename_stream << "time_svd_" << "factors-" << K
//                     << "-err-" << probe_err << "-epochs-" << epochs << "-init-" << init;
//                 string s = filename_stream.str();
//                 cout << "Writing to: " << s << endl;
//
//                 vector<double> preds = m.predict();
//                 writeToFile(s, preds);
//             }
//         }
//     }
// }

    for (double eta = 0.02; eta < 0.035; eta += 0.005) {
                for (double reg = 0.005; reg < 0.04; reg += 0.005) {
                Model m(458293, 17770, 40, eta, reg, train_filename, test_filename, valid_filename);
                cout << " Training model with parameters (K, eta, reg) "
                             "40 " << eta << " " << reg <<  endl;
                m.train();
                double probe_err = m.validErr();
                stringstream filename_stream;
                filename_stream << "svd_plus" << "factors-40"<< "-eta-" << eta << "-reg-" << reg
                    << "-err-" << probe_err <<".txt";
                string s = filename_stream.str();
                cout << "Writing to: " << s << endl;

                vector<double> preds = m.predict();
                writeToFile(s, preds);
            }
    }
}

int main() {
    gridSearch();
    return 0;
}
