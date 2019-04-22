#include <vector>
#include <stdio.h>

#include "matrix_factor_bias.h"

using namespace std;

int main() {
    string filename = "/Users/vigneshv/code/CS156b-Netflix/matrix_factor/mu/all.dta";

    Model m(458293, 17770, 20, 0.01, 0.1, filename, 3.84358);
    m.train();

    return 0;
}
