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

    while (valid_data.hasNext()) {
        NetflixData d = valid_data.nextLine();
        int rating = d.rating;

        double pred;
        pred << predictions;

        file << d.user << " " << d.movie << " " << " " 
            << d.date << " " (double) rating - pred << endl;
    }

    file.close();
}

int main() {
    string train_filename = "/home/ubuntu/CS156b-Netflix/rbm/rbm_train.txt";
    string prediction_filename = "/home/ubuntu/CS156b-Netflix/data/train2.dta";

    get_residuals(prediction_filename, train_filename);
}
