

#include "DestinTreeManager.h"
#include "DestinTreeIteratorCallback.h"
#include "DestinNetworkAlt.h"
#include "VideoSource.h"
#include <ostream>

using std::endl;

class AtomGenerator : public DestinTreeIteratorCallback {
    std::ostream& output;
public:

    AtomGenerator(std::ostream& output):output(output){}

    virtual void callback(const Node& node, int child_position){
        output << "(hasCentroid " << node.nIdx << " " << node.winner << " )" << endl;

        int layer = node.layer;
        Destin * destin = node.d;

        if(node.d->hasMultiParents[layer]){
            for(int p = 0 ; p < node.nParents ; p++){
                Node * parent = node.parents[p];
                if(parent != NULL){
                    output << "(has" << p << "ParentCentroid " << node.nIdx << " " << parent->winner << " )" << endl;
                    output << "(hasParent " << node.nIdx << " " << parent->nIdx << " )" << endl;
                }
            }
        } else {
            if(node.firstParent != NULL){
                int parent_num = 3 - child_position;
                output << "(has" << parent_num << "ParentCentroid " << node.nIdx << " " << node.firstParent->winner << " )" << endl;
                output << "(hasParent " << node.nIdx << " " << node.firstParent->nIdx << ")" << endl;
            }
        }

        int layerWidth = destin->layerWidth[layer];
        Node * neighbor;
        if(node.row > 0){
            neighbor = GetNodeFromDestin(destin, layer, node.row - 1, node.col);
            output << "(hasNorthNeighborCentroid " << node.nIdx << " " << neighbor->winner << " )" << endl;
            output << "(hasNorthNeighbor " << node.nIdx << " " << neighbor->nIdx << " )" << endl;
        }

        if(node.row < layerWidth - 1){
            neighbor = GetNodeFromDestin(destin, layer, node.row + 1, node.col);
            output << "(hasSouthNeighborCentroid " << node.nIdx << " " << neighbor->winner << " )" << endl;
            output << "(hasSouthNeighbor " << node.nIdx << " " << neighbor->nIdx << " )" << endl;
        }

        if(node.col > 0){
            neighbor = GetNodeFromDestin(destin, layer, node.row, node.col - 1);
            output << "(hasWestNeighborCentroid " << node.nIdx << " " << neighbor->winner << " )" << endl;
            output << "(hasWestNeighbor " << node.nIdx << " " << neighbor->nIdx << " )" << endl;
        }

        if(node.col < layerWidth - 1){
            neighbor = GetNodeFromDestin(destin, layer, node.row, node.col + 1);
            output << "(hasEastNeighborCentroid " << node.nIdx << " " << neighbor->winner << " )" << endl;
            output << "(hasEastNeighbor " << node.nIdx << " " << neighbor->nIdx << " )" << endl;
        }
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
