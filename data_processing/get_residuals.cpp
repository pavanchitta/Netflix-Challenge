#include <string>
#include <iostream>
#include <fstream>
#include "data.h"

using namespace std;

void get_residuals(string prediction_filename, string train_filename) {
    Data train_data = Data(train_filename);
	ofstream file;

    ifstream predictions(prediction_filename);

    file.open("residuals.dta");

    while (train_data.hasNext()) {
        string temp;
        getline(predictions, temp);
        double pred = std::stod(temp);

        NetflixData d = train_data.nextLine();
        int rating = d.rating;

        file << d.user << " " << d.movie << " "
            << d.date << " " << (double) rating - pred << endl;
    }

    file.close();
}

int main() {
    string prediction_filename = "/Users/vigneshv/code/CS156b-Netflix/data/rbm_train.txt";
    string train_filename = "/Users/vigneshv/code/CS156b-Netflix/data/train2.dta";

    get_residuals(prediction_filename, train_filename);
}
