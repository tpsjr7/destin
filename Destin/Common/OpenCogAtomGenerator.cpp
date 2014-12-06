

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



class DestinGraphNodePrinter : public DestinGraphIteratorCallback {
    std::ostream & out;

public:
    DestinGraphNodePrinter(std::ostream & out)
        :out(out){}

    virtual void callback(const Node& node, const bool isBottom, uint * nodeIdToGraphNodeId){
        out << "n " << nodeIdToGraphNodeId[node.nIdx] << " L_" << node.layer << "_W_" << node.winner << endl;
        return;
    }
};

class DestinGraphEdgePrinter : public DestinGraphIteratorCallback {
    std::ostream & out;

public:
    DestinGraphEdgePrinter(std::ostream & out)
        :out(out){}

    virtual void callback(const Node& node, const bool isBottom, uint * nodeIdToGraphNodeId){
        if(isBottom){
            return;
        }

        const int nChildren = node.nChildren;
        for(int i = 0 ; i < nChildren; i++){
            out << "e " << nodeIdToGraphNodeId[node.nIdx] << " " << nodeIdToGraphNodeId[node.children[i]->nIdx] << " child_edge_" << i << endl;
        }
        return;
    }
};

void usage(char ** argv){
    cout << argv[0] << " -t [a=atoms, g=graph] -l [layers 2 to 7] -tc [tree count] -w widths -c [centroids] -iw [img width]" << endl;
}

struct ArgConfig {

    ArgConfig(){
        type = "g";
        uint the_width[] = {4,2,1};
        widths = vector<uint>(the_width, the_width + 3);
        layers = 3;
        treeCount = 1;
        uint the_centroids[] = {5,5,5};
        centroids = vector<uint>(the_centroids, the_centroids + 3);
        imgWidth = 16;
    }

    string type;
    vector<uint> widths;
    vector<uint> centroids;
    int layers;
    int treeCount;
    int imgWidth;
};

// convert string of comma seperated numbers into an array
vector<uint> splitNumbers(char * strNums){
    char * c;
    c = strtok(strNums,",");
    vector<uint> nums;
    while(c != NULL){
        nums.push_back(atoi(c));
        c = strtok(NULL, ",");
    }

    return nums;
}

ArgConfig parseArgs(int argc, char ** argv){
    int arg = 1;

    std::vector<char *> args(argv, argv + argc);

    ArgConfig config;
    while(arg < argc){
        string argString(args.at(arg));

        arg++;
        if(argString == "-t"){
            config.type = string(args.at(arg));
        } else if(argString == "-w"){
            config.widths = splitNumbers(args.at(arg));
        } else if(argString == "-c"){
            config.centroids = splitNumbers(args.at(arg));
        } else if(argString == "-l"){
            config.layers = atoi(args.at(arg));
        } else if(argString == "-tc"){
            config.treeCount =  atoi(args.at(arg));
        } else if(argString == "-iw"){
            config.imgWidth = atoi(args.at(arg));
        }
        arg++;
    }

    return config;
}

int main(int argc, char ** argv){

    if(argc == 1){
        usage(argv);
        return 0;
    }

    ArgConfig config = parseArgs(argc, argv);

    string mode = config.type;

    int layers = config.layers;


    uint * counts = &config.centroids[0];

    int width = config.imgWidth;
    int trees = config.treeCount;

    DestinNetworkAlt dna(width, layers, counts, true, &config.widths[0]);

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

    DestinGraphNodePrinter np(std::cout);
    DestinGraphEdgePrinter ep(std::cout);

    for(int i = 0 ; i < trees ; i++){
        vs.grab();
        dna.doDestin(vs.getOutput());

        if(mode == "a"){
            tm.iterateTree(ag);
        } else if(mode == "g") {
            tm.iterateGraph(np);
            tm.iterateGraph(ep);
            cout << endl;
        }
    }

    return 0;
}
