#!/bin/sh

destin_alt=`cd ../../DavisDestin ; pwd`
java_destin=`pwd`
the_path="${destin_alt}:${java_destin}"

the_cp="./groovy_proj/out/production/groovy_proj/:./build/classes"

#make sure you've downloaded groovy

~/groovy-2.*/bin/groovysh -classpath "$the_cp"  -Djava.library.path="$the_path"
