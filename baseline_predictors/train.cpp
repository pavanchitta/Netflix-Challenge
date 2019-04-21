#include "baseline_predictor.h"

int main() {

    string filename = "/Users/pavanchitta/CS156b-Netflix/data/um/all.dta";
    Model model = Model(458293, 17770, filename, 0.01, 40);
    cout << "Starting train process " << endl;
    model.train();

    return 0;
}
