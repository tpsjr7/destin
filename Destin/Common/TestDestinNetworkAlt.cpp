#include "unit_test.h"
#include "DestinNetworkAlt.h"
#include <ostream>

using namespace std;

int regressionTest();
int testSaveAndReload();

int main(int argc, char ** argv){
    RUN(regressionTest);
    RUN(testSaveAndReload);
    UT_REPORT_RESULTS();
    return 0;
}

int regressionTest(){
    dst_ut_srand(12345); // seed our test random number generator
    srand(101); // seed the normal random number generator

    uint centroids[4] = {4,4,4,4};
    DestinNetworkAlt dna(W32, 4,  centroids, true);

    int nImages = 10;
    float ** images = makeRandomImages(32*32, nImages);
    int iterations = 50;
    SetLearningStrat(dna.getNetwork(), CLS_FIXED);
    dna.setFixedLearnRate(0.1);

    for(int i = 0 ; i < iterations ; i++ ){
        dna.doDestin(images[i % nImages]);
    }

    cout << dna.getNode(3,0,0)->beliefMal[0] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[1] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[2] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[3] << endl;

    freeRandomImages(images, nImages);
    return 0;
}

int testSaveAndReload(){
    uint centroids[4] = {3,4,2,3};
    DestinNetworkAlt dna(W32, 4,  centroids, true);

    int nImages = 10;
    float ** images = makeRandomImages(32*32, nImages);
    int iterations = 100;
    SetLearningStrat(dna.getNetwork(), CLS_FIXED);
    dna.setFixedLearnRate(0.1);

    for(int i = 0 ; i < iterations ; i++ ){
        dna.doDestin(images[i % nImages]);
    }

    dna.save("test-save1.dst");

    for(int i = 0 ; i < iterations ; i++ ){
        dna.doDestin(images[i % nImages]);
    }

    // Save the beliefs
    // get beliefs for layers
    float beliefsLayer0[64 * 3];
    float beliefsLayer1[16 * 4];
    float beliefsLayer2[4 * 2];
    float beliefsLayer3[1 * 3];
    dna.getLayerBeliefs(0, beliefsLayer0);
    dna.getLayerBeliefs(1, beliefsLayer1);
    dna.getLayerBeliefs(2, beliefsLayer2);
    dna.getLayerBeliefs(3, beliefsLayer3);

    assertNoNans(beliefsLayer0, 64*3);
    assertNoNans(beliefsLayer1, 64*4);
    assertNoNans(beliefsLayer2, 64*2);
    assertNoNans(beliefsLayer3, 64*3);

    DestinNetworkAlt reloaded("test-save1.dst");

    for(int i = 0 ; i < iterations ; i++ ){
        reloaded.doDestin(images[i % nImages]);
    }

    float newbeliefsLayer0[64 * 3];
    float newbeliefsLayer1[16 * 4];
    float newbeliefsLayer2[4 * 2];
    float newbeliefsLayer3[1 * 3];
    reloaded.getLayerBeliefs(0, newbeliefsLayer0);
    reloaded.getLayerBeliefs(1, newbeliefsLayer1);
    reloaded.getLayerBeliefs(2, newbeliefsLayer2);
    reloaded.getLayerBeliefs(3, newbeliefsLayer3);

    //should be in the same state
    assertFloatArrayEquals(beliefsLayer0, newbeliefsLayer0, 64*3);
    assertFloatArrayEquals(beliefsLayer0, newbeliefsLayer0, 16*4);
    assertFloatArrayEquals(beliefsLayer0, newbeliefsLayer0, 4*2);
    assertFloatArrayEquals(beliefsLayer0, newbeliefsLayer0, 1*3);

    freeRandomImages(images, nImages);

    return 0;
}

int testAddCentroid(){
    uint centroids[4] = {4,4,4,4};
    DestinNetworkAlt dna(W32, 4,  centroids, true);
    return 0;
}
