

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

class DestinGraphPrinter : public DestinGraphIteratorCallback {
protected:
    std::ostream & out;
    char labelMode;
    int layers;
    const int layerSpace;
public:
    DestinGraphPrinter(std::ostream & out, char labelMode, int layers)
        :out(out), labelMode(labelMode), layers(layers), layerSpace(1000000 / layers){}

};

class DestinGraphNodePrinter : public DestinGraphPrinter {
public:
    DestinGraphNodePrinter(std::ostream & out, char labelMode, int layers)
        :DestinGraphPrinter(out, labelMode, layers) {}

    virtual void callback(const Node& node, const bool isBottom, uint * nodeIdToGraphNodeId){
        out << "v " << nodeIdToGraphNodeId[node.nIdx];
        if(labelMode == 'g'){
            out << " " << (layerSpace * node.layer + node.winner) << endl;
        } else if(labelMode == 'm'){
            out << " L_" << node.layer << "_W_" << node.winner << endl;
        }
        return;
    }
};

class DestinGraphEdgePrinter : public DestinGraphPrinter {
public:
    DestinGraphEdgePrinter(std::ostream & out, char labelMode, int layers)
         :DestinGraphPrinter(out, labelMode, layers) {}

    virtual void callback(const Node& node, const bool isBottom, uint * nodeIdToGraphNodeId){
        if(isBottom){
            return;
        }

        const int nChildren = node.nChildren;
        if(labelMode == 'g'){
            for(int i = 0 ; i < nChildren; i++){
                out << "e " << nodeIdToGraphNodeId[node.nIdx] << " " << nodeIdToGraphNodeId[node.children[i]->nIdx] << " " << i << endl;
            }
        } else if(labelMode == 'm'){
            for(int i = 0 ; i < nChildren; i++){
                out << "e " << nodeIdToGraphNodeId[node.nIdx] << " " << nodeIdToGraphNodeId[node.children[i]->nIdx] << " child_edge_" << i << endl;
            }
        }
        return;
    }
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

void usage(char ** argv){
    cout << argv[0] << " --mode [a=atoms, g=graph] --layers [layers] --n-out [count] " << endl
                    << "--widths widths --cents [centroids] --img-width [img width]" << endl
                    << "< --load [filename] > < --save [filename] > < --hide-video > < --train-frames [frames=50] >" << endl
                    << "< --out filename > < --label-mode=m|g >" << endl
                       ;
}

struct ArgConfig {

    ArgConfig(){
        uint the_centroids[] = {5,5,5};
        centroids = vector<uint>(the_centroids, the_centroids + 3);

        imgWidth = 16;
        labelMode = 'g';
        layers = 3;
        load = "";
        mode = "g";
        outputFile = "";
        showVid = true;
        trainFrames = 50;
        treeCount = 1;
        uint the_width[] = {4,2,1};
        widths = vector<uint>(the_width, the_width + 3);
    }

    vector<uint>    centroids;
    int             imgWidth;
    char            labelMode;
    int             layers;
    string          load;
    string          mode;
    string          outputFile;
    string          save;
    bool            showVid;
    int             trainFrames;
    int             treeCount;
    vector<uint>    widths;
};

ArgConfig parseArgs(int argc, char ** argv){
    int arg = 1;

    std::vector<char *> args(argv, argv + argc);

    ArgConfig config;
    while(arg < argc){
        string argString(args.at(arg));

        arg++;
        if(argString == "--mode"){
            config.mode = string(args.at(arg));
        } else if(argString == "--widths"){
            config.widths = splitNumbers(args.at(arg));
        } else if(argString == "--cents"){
            config.centroids = splitNumbers(args.at(arg));
        } else if(argString == "--layers"){
            config.layers = atoi(args.at(arg));
        } else if(argString == "--n-out"){
            config.treeCount =  atoi(args.at(arg));
        } else if(argString == "--img-width"){
            config.imgWidth = atoi(args.at(arg));
        } else if(argString == "--load"){
            config.load = args.at(arg);
        } else if(argString == "--save"){
            config.save = args.at(arg);
        } else if(argString == "--train-frames"){
            config.trainFrames = atoi(args.at(arg));
        } else if(argString == "--hide-video"){
            config.showVid = false;
            arg--;//doesn't take a value
        } else if(argString == "--out"){
            config.outputFile = args.at(arg);
        } else if(argString == "--label-mode"){
            char mode = args.at(arg)[0];
            switch(mode){
                case 'm':
                case 'g':
                    break;
                default:
                    cerr << "Invalid label mode." << endl;
                    exit(1);
            };

            config.labelMode = mode;
        } else {
            cerr << "Invalid argument " << argString << endl;
            exit(1);
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

    ostream * out_stream;
    ofstream fs;

    if(config.outputFile != ""){
        fs.open(config.outputFile.c_str());
        if(!fs.is_open()){
            stringstream mess ; mess << "Could not open file " << config.outputFile << endl;
            throw std::runtime_error(mess.str());
        }
        out_stream = &fs;
    } else {
        out_stream = &std::cout;
    }

    int width = config.imgWidth;
    DestinNetworkAlt * dna = NULL;

    VideoSource vs(false,"../Bindings/Python/moving_square.m4v");
    vs.setSize(width,width);
    if(config.showVid){
        vs.enableDisplayWindow();
    }

    if(config.load == ""){
        dna  = new DestinNetworkAlt(width, config.layers,  &config.centroids[0], true, &config.widths[0]);
        cout << "Training " << config.trainFrames << " frames..." << endl;

        for(int i = 0 ; i < config.trainFrames ; i++){
            vs.grab();
            dna->doDestin(vs.getOutput());
            if(i % 10 == 0){
                cout << i << " Q: " << dna->getQuality(config.layers - 1) << endl;
            }
        }
        cout << endl << "Done training." << endl;
    } else {
        dna = new DestinNetworkAlt(config.load.c_str());
        cout << "Loaded: " << config.load << endl;
        int size = dna->getNetwork()->inputImageSize;
        if(width * width != size){
            int new_width = (int)sqrt(size);
            cout << "Input image size overridden to " << new_width << " pixels wide from " << width << endl;
            width = new_width;
            vs.setSize(width,width);
        }
    }

    DestinTreeManager tm(*dna, 0, config.labelMode);

    AtomGenerator ag(*out_stream);
    DestinGraphNodePrinter np(*out_stream, config.labelMode, config.layers);
    DestinGraphEdgePrinter ep(*out_stream, config.labelMode, config.layers);

    dna->setIsTraining(false);

    for(int i = 0 ; i <  config.treeCount ; i++){

        if(!vs.grab()){
            vs.grab(); // try again if at the end of the video
        }
        dna->doDestin(vs.getOutput());

        if(config.mode == "a"){
            tm.iterateTree(ag);
        } else if(config.mode == "g") {
            if(config.labelMode == 'g'){
                *out_stream << "# t 1" << endl;
            }
            tm.iterateGraph(np);
            tm.iterateGraph(ep);
            if(i < config.treeCount - 1){
                *out_stream << endl; // avoid an empty line at the end of the file
            }
        }
    }

    if(config.save != ""){
        dna->save(config.save.c_str());
    }
    delete dna;
    if(config.outputFile != ""){
        fs.close();
    }
    return 0;
}
