
import javadestin.*
import javadestin.SupportedImageWidths

Main.init()

train = {
    for (i in 0..<train_frames){
        if(!ims.grab()){
            continue
        }
        dn.doDestin(ims.getOutput())
        dn.printBeliefGraph(7,0,0)
    }
    dn.save(dst_save_file)
    return
}
            
getTrees = {
    dn.setIsPOSTraining(false) //// freeze training
    for (i in 0..<mine_frames){
        if (!ims.grab()){
            println "could not grab frame"
            continue;
        }
            
        dn.doDestin(ims.getOutput())
        tm.addTree()
    }
    tm.timeShiftTrees()
}

        
displayMinedTree = { index ->
    if (index >= found){
        println "out of bounds"
        return
    }
    tm.displayMinedTree(index)
    DESTIN_MODULE.waitKey(200)
}

displayAllTrees = {
    for (i in 0..<found){
        println "Showing ${i} of ${found - 1}"
        tm.printFoundSubtreeStructure(i)
        tm.displayFoundSubtree(i)
        DESTIN_MODULE.waitKey(2000)
    }
}
        

displayCentroidImages = { layer ->
    dn.displayLayerCentroidImages(layer)
    DESTIN_MODULE.waitKey(200)
}


// matches subtrees to their destin image
matchSubtrees = {
    for (i in 0..<tm.getFoundSubtreeCount()){
        matches = tm.matchSubtree(i)
        println "found tree ${i} matches:"
        for(int j in 0..<matches.size()){
            print matches.get(j)+" "
        }
        println()
    }
}

mkdir = { path ->
    def f = new File(path)
    f.mkdirs()
}

saveResults = { run_id ->
    def out_dir="tree_mining_runs/"+run_id+"/"
    mkdir(out_dir)
    for (l in 0..<layers){
        dn.saveLayerCentroidImages(l, out_dir+"layer_"+l+".png");
    }

    def fw = new FileWriter(out_dir+"matches.txt")

    for(t in 0..<tm.getFoundSubtreeCount()){
        tm.saveFoundSubtreeImg(t, out_dir+"subtree_${t}.png")
        IntVector matches = tm.matchSubtree(t)
        fw.append("tree //${t} matches input image: \n")
        for(int j in 0..<matches.size()){
            fw.append(matches.get(j)+" ")
        }
        fw.append("\n")
    }
    fw.close()

    fw = new FileWriter(out_dir+"treepath.txt")
    gw = new FileWriter(out_dir+"treepath_structure.txt")

    for(t in 0..<tm.getFoundSubtreeCount()){
        fw.write("tree //${t}: ${tm.getFoundSubtreeAsString(t)}\n")
        gw.write("tree //${t}:\n${tm.getFoundSubtreeStructureAsString(t)}\n")
    }
    fw.close()
    gw.close()

    new File(out_dir+"network_save.bin").withOutputStream { w ->
        w << new File(dst_save_file).bytes
    }

    letters.eachWithIndex { def letter, int i ->
        new File(out_dir+"input_img_${i}.png").bytes = new File("${img_path}${letter}.png").bytes
    }
}

//// Init destin
centroids = [2, 2, 4, 8, 32, 16, 8, 3] as long[]
layers = centroids.size()
dn = new DestinNetworkAlt(SupportedImageWidths.W512, layers, centroids, true)
dn.setBeliefTransform(BeliefTransformEnum.DST_BT_P_NORM)
//dn.setBeliefTransform(pd.DST_BT_NONE)
//dn.setBeliefTransform(pd.DST_BT_BOLTZ)

uniform_temp = 2.0
temperatures = []
for( i in 0..<8){
    temperatures.push(uniform_temp)
}
//temperatures = [5, 5, 10, 20, 40, 20, 16, 6]

dn.setTemperatures(temperatures as float[])
dn.setIsPOSTraining(true)
dn.setFixedLearnRate(0.01)

dn.setCentImgWeightExponent(4)

ims = new ImageSouceImpl(512, 512)

letters = "+LO"

img_path = "/Users/tedsandersjr/Dropbox/destin/treeimgs/"
for(l in letters){
    ims.addImage("${img_path}${l}.png")
}

dst_save_file="minedtree.dst"

//// setup tree manager
bottom_layer = 4
tm = new DestinTreeManager(dn, bottom_layer)


mine_frames = letters.length() + layers - 1
train_frames = 1600

//train()
dn.load(dst_save_file)


getTrees()    

support = 2
found = tm.mine(support)
println "found ${found} trees"

matchSubtrees()
displayCentroidImages(7)    

saveResults("2")
println "click on tree images to continue"
displayAllTrees()
