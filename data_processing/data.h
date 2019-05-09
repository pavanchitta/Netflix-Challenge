#include <string>
#include <vector>
#include <fstream>

#ifndef DATA_H
#define DATA_H

using namespace std;

typedef struct {
    int32_t user;
    int16_t movie;
    int16_t date;
    int8_t rating;
} NetflixData;

typedef struct {
    NetflixData *arr;
    int size;
} DataCache;

class Data {
    private:
        string filename;
        char *overhang;
        ifstream* csv;

        DataCache cache; 

        int vec_inx;
        void fill_vector ();
    public:
        Data (string f);
        NetflixData nextLine();
        bool hasNext();
        void reset();
        ~Data();
};

#endif
