The openCogAtomGenerator binary will generate some atoms to be imported into the OpenCog atomspace.

It builds a 7 layer hierarchy, which works on a 256x256 pixel video image "../Bindings/Python/moving_square.avi"

It trains on the video for a really short time, 50 frames. The goal is to simply generate some data so not much effort is spent in training the network.

Then a DestinTreeManager is used to iterate through all the nodes in the DeSTIN network, and uses the "AtomGenerator" callback to generate some scheme atoms in text format.

> cd Destin/Common/

> ./openCogAtomGenerator > atoms.txt

Start the OpenCog server in a new console tab:

> cd ~/opencog/build; ./opencog/server/cogserver

Back in Destin/Common/ directory:

> (echo "scm" ; cat predicates.scm atoms.txt | head -10000 ) | telnet localhost 17001

The head -10000 is there to limit the amount of Atoms, as it takes too much memory currently.
