

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
                    output << "(has" << p << "Parent "         << node.nIdx << " " << parent->nIdx << " )" << endl;
                }
            }
        } else {
            if(node.firstParent != NULL){
                int parent_num = 3 - child_position;
                output << "(has" << parent_num << "ParentCentroid " << node.nIdx << " " << node.firstParent->winner << " )" << endl;
                output << "(has" << parent_num << "Parent "  << node.nIdx << " " << node.firstParent->nIdx << " )" << endl;
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

void usage(char ** argv){
    cout << argv[0] << " [layers 2 to 7] [tree count]" << endl;
}

int main(int argc, char ** argv){

    if(argc != 3){
        usage(argv);
        return 0;
    }

    int layers = atoi(argv[1]);

    uint counts[] = {5,5,5,5,5,5,5,5,5,5};

    SupportedImageWidths width;
    switch (layers) {
        case 7:
            width = W256;
            break;
        case 6:
            width = W128;
            break;
        case 5:
            width = W64;
            break;
        case 4:
            width = W32;
            break;
        case 3:
            width = W16;
            break;
        case 2:
            width = W8;
            break;
        default:
            usage(argv);
            return 0;
    }

    int trees = atoi(argv[2]);

    DestinNetworkAlt dna(width, layers, counts, true);

    //dna.load("../Bindings/Python/square.dst");

    VideoSource vs(false,"../Bindings/Python/moving_square.avi");
    vs.setSize(width,width);
    vs.enableDisplayWindow();
    for(int i = 0 ; i < 50 ; i++){
        vs.grab();
        dna.doDestin(vs.getOutput());
    }

    DestinTreeManager tm(dna, 0);
    AtomGenerator ag(std::cout);

    for(int i = 0 ; i < trees ; i++){
        vs.grab();
        dna.doDestin(vs.getOutput());
        tm.iterateTree(ag);
    }

    return 0;
}
