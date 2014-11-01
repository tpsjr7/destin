import javadestin.DestinNetworkAlt;
import javadestin.SupportedImageWidths;
import javadestin.VideoSource;

/**
 * Created by tedsandersjr on 10/23/14.
 */
public class Main {
    static {
        init();
    }

    public static void init(){
        System.load("/Users/tedsandersjr/destin/Destin/Bindings/Java/libdestinjava.jnilib");
        System.load("/Users/tedsandersjr/destin/Destin/DavisDestin/libdestinalt.dylib");
    }
    public static void main(String [] args){
        DestinNetworkAlt dna = new DestinNetworkAlt(SupportedImageWidths.W512, 7, new long[]{5,5,5,5,5,5,5}, true);
        VideoSource vs = new VideoSource(true, "");
        vs.setSize(512, 512);
        vs.enableDisplayWindow();
        for(int i = 0 ; i < 20 ; i++){
            vs.grab();
            dna.doDestin(vs.getOutput());
            dna.printBeliefGraph(6, 0, 0);
        }
    }
}
