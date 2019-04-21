#include <string>
#include <vector>
#include <fstream>

using namespace std;

class Data {
    private:
        string filename;
        ifstream* csv;
        vector<vector<int> > vectors;

        int vec_inx;
        void fill_vector ();
    public:
        Data (string f);
        vector<int> nextLine();
        bool hasNext();
        void reset();
};
