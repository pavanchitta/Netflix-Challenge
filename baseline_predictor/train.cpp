#include "baseline_predictor.h"

int main() {

    string filename = "/Users/rahilbathwal/Library/Mobile Documents/com~apple~CloudDocs/College/Spring 2019/CS 156b/CS156b-Netflix/data/um/train.dta";
    BaselinePredictor model = BaselinePredictor(458293, 17770, filename, 0.01, 40);
    cout << "Starting train process " << endl;
    model.train();

    return 0;
}
