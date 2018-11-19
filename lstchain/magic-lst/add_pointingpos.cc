//=======================
/* This is to add PointingPos container to the _Y_ files 
 Author(s): David Fidalgo, Lab Saha
*/
//========================
#include "MReadMarsFile.h"
#include "MParList.h"
#include "MTaskList.h"
#include "MWriteRootFile.h"
#include "MEvtLoop.h"

#include "MPointingPos.h"
#include "MPointingPosCalc.h"
#include "MPointingPosInterpolate.h"
#include "MObservatory.h"
#include "MStarguiderCalibration.h"
#include "MReadReports.h"

#include "TString.h"

#include <iostream>


int add_pointingpos(const char* datafile,const char* outdir) {
//int main(char * datafile[]) {
    TString input = datafile;
//  Edit the path of the output directory
//    TString outputDirectory = "../../data_ctapipe/out";
    TString outputDirectory = outdir;
    Bool_t IsUseStarguider = kTRUE;


    // read in everything
    MReadReports readreal;

    readreal.AddTree("Events", "MTime.", kTRUE);
    readreal.AddTree("Drive");
    readreal.AddTree("Pedestals");
    readreal.AddTree("CC");
    readreal.AddTree("Pyrometer");
    readreal.AddTree("Starguider");
    readreal.AddTree("Trigger");
    readreal.AddTree("Runbook");
    readreal.AddTree("AMC");
    readreal.AddTree("DT");
    readreal.AddTree("DAQ");
    readreal.AddTree("SUMO");
    readreal.AddTree("Camera");
    readreal.AddTree("Lidar") ;
    readreal.AddTree("Laser");
    readreal.AddTree("Weather");
    readreal.AddTree("SumTrigger");
    readreal.AddTree("L3T");

    readreal.AddFile(input.Data());


    // Parameter list: contains all parameter containers (input/output)
    // The reader task will automatically put in the parameter list the 
    // containers which are read-in.
    MParList parlist;

    // Task list: ordered list of all the tasks to be performed in the event loop
    MTaskList tasklist;

    // The task list itself has to be put into the parameter list:
    parlist.AddToList(&tasklist);

    MStarguiderCalibration fCalibStar;
    if (IsUseStarguider)
      parlist.AddToList(&fCalibStar);

    // Now add the reading task to the task list:
    tasklist.AddToList(&readreal);

    MPointingPos pos;
    parlist.AddToList(&pos);
    MObservatory obs;
    parlist.AddToList(&obs);

    MPointingPosInterpolate pextr;

    pextr.AddFile(input.Data()); 
    pextr.SetUseStarguider(IsUseStarguider);
    tasklist.AddToList(&pextr);
    
    // Now a task to write out an output file, containing the usual MARS trees (Events, RunHeaders) but 
    // with only the data we want inside.
    // First argument of constructor is compression level (see root Manual). 
    // The second argument is the "rule" to obtain the name of the output file from the name of the input file. 
    // Usually the name is kept unchanged except for the "tag" between underscores, for instance *_I_*.root ==> *_Q_*.root
    // when we run melibea over star files. In this example we replace the _I_ of star files by _H_
    // Before the "{" in the second argument we set the output directory
    // The third argument means the output files will be overwritten if already existing.
    MWriteRootFile write(2, Form("%s{s/_Y_/_Y_}", outputDirectory.Data()), "RECREATE");

    // Now add the containers you want to keep in the output, specifying the tree to which they have to be added (2nd argument).
    // The containers may be either newly created ones or containers already existing in the input file (which will be 
    // present in the parameter list)
    write.AddCopySource("Events");
    write.AddContainer("MPointingPos", "Events");
    write.AddContainer("MRawEvtHeader",  "Events");
    write.AddContainer("MTime", "Events");
    write.AddContainer("MCerPhotEvt", "Events");
    write.AddContainer("MTriggerPattern", "Events");
    write.AddContainer("MArrivalTime", "Events"); 
    
    // NOTE: in the containers, like MHillas in this case, which are simply "copied" to the output from the input file,
    // only the enabled branches (see "read" task at beginning of code) will be filled. The rest will just contain the
    // default values!

    // Now add the writing task to the tasklist:
    tasklist.AddToList(&write);

    // The Mars event loop. One has to specify which is its parameter list:
    MEvtLoop evtloop;
    evtloop.SetParList(&parlist);

    //
    // We execute the loop: 
    //
    // - The PreProcess function of All the tasks in the tasklist will be executed before doing anything else
    // - Then the Process function of all tasks is executed for each event in the Events tree of the Mars files,
    //   except if a filter avoids it
    // - The ReInit function (if it exists) of each task is executed every time a new Mars file is opened. This
    //   is needed because some parameters are written in the RunHeaders tree (which has one entry per file) and 
    //   hence they are read every time a new file is opened. So in the ReInit function of a task we can update
    //   data members which depend on information held in RunHeaders
    // - At the end of the loop the PostProcess function of all tasks in the task list is called.
    //
    if (!evtloop.Eventloop())
        return -1;

    // Print statistics of the event loop:
    tasklist.PrintStatistics();

    return 0;
}
