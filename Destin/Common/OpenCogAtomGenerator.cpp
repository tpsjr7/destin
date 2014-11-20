

#include "DestinTreeManager.h"
#include "DestinTreeIteratorCallback.h"
#include "DestinNetworkAlt.h"
#include "VideoSource.h"
#include <ostream>

class AtomGenerator : public DestinTreeIteratorCallback {
    std::ostream& output;
public:

    AtomGenerator(std::ostream& output):output(output){}

    virtual void callback(const Node& node, int child_position){
        output << "(hasCentroid " << node.nIdx << " " << node.winner << " )" << std::endl;
    }

};

int main(int argc, char ** argv){

    uint counts[] = {5,5,5,5,5,5,5};

    DestinNetworkAlt dna(W256, 7, counts, true);

    VideoSource vs(true,"");
    vs.setSize(256,256);
    vs.enableDisplayWindow();
    for(int i = 0 ; i < 50 ; i++){
        vs.grab();
        dna.doDestin(vs.getOutput());
    }

    DestinTreeManager tm(dna, 0);
    AtomGenerator ag(std::cout);
    tm.iterateTree(ag);

    return 0;
}
