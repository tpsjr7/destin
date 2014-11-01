/**
 * Created by tedsandersjr on 10/22/14.
 */

import javadestin.DestinNetworkAlt
import javadestin.SupportedImageWidths
import javadestin.VideoSource
import javadestin.*

class Experiment {
    static{
        Main.init();
    }

    def centroids = [5,5,5,5,5,5,2] as long[]
    def dn = new DestinNetworkAlt(SupportedImageWidths.W512, 7, centroids, true)
    def vs = new VideoSource(true, "")

    void init(){
        vs.setSize(512, 512)
        vs.enableDisplayWindow()
        go()
    }

    void go(int frames = 100, int layer=0){
        for(int i in 1..frames){
            if(vs.grab()){
                dn.doDestin(vs.getOutput());

                dn.printBeliefGraph(6,0,0)
            }
            if(i%10 == 0){
                dn.updateCentroidImages()
                dn.displayLayerCentroidImages(layer)
            }
        }
    }

    void go(int frames = 100, Closure cb){
        for(int frame in 1..frames){
            if(vs.grab()){
                dn.doDestin(vs.getOutput())
                cb(frame)
            }
        }
    }
}

e = new Experiment()
dn = e.dn

dn.metaClass.setIsTraining = {dn.isTraining(it)}
dn.metaClass.getIsTraining = {return dn.isTraining()}

bottom_layer = 2
tree_add_frequency = 10
tm = new DestinTreeManager(e.dn, bottom_layer)

e.init()
e.go(300)
go = {e.go(it)}

treeCollector = { frame ->
    if(frame % tree_add_frequency == 0){
        tm.addTree()
        println "Added tree. Has ${tm.getAddedTreeCount()}"
    }
}

dn.isTraining(false)

e.go(300, treeCollector)

mineOnThread = { support ->
    Thread t = new Thread({
        println "Starting mining."
        println "Found ${tm.mine(support)}"
    });
    t.start();
    return t;
}

support = 10
mt = mineOnThread(support)

