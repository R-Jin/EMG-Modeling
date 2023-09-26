<style type="text/css">
    #questions { list-style-type: upper-alpha; }
</style>

# EMG-Modeling

## Q1

![Illustration](https://cdn.discordapp.com/attachments/971680379993989130/1156167058707456062/image.png?ex=6513fbfb&is=6512aa7b&hm=f84518df38f4b45175e0b14fb48084881b3b3fc18ae31211e848cb933d730b61& "Abstraction of the data")

<ol type="a" id="questions">
  <li>
    <h3>Create the trains of potentials corresponding to each unit (8 trains in total).</h3>
    <p>
      We prepared a zero-array with the length of the samples within the given duration for each train. We iterate through each index in firing_sample and point at the index. From that point, we go through the action potential and add correspondingly to each step (We slice the array and copy the whole action potential in place).
    </p>
  </li>
  <li>
    <h3>How many samples do each action potential train contain? Why? We expect a well-reasoned answer based on the theories discussed in class.</h3>
    <p>Each action potential train contains 200.000 samples. This is due to the duration of the signal (20s), and our sample rate().</p>
    </li>
  <li>
  <h3>Plot 1 of the 8 action potential trains as a function of time (therefore you should have 0-20 s 
    in the time axis). In addition, plot the same action potential train in the time interval 10-10.5 
    s.
    Note: All axes must be labelled. The unit for the time axis should be s (seconds); the unit for 
    the amplitude of the action potentials is not provided and you should indicate A.U. (which 
    stands for arbitrary unit).</h3>
    <p></p>
</li>
<li>
  <h3>Comment on the procedure you have followed to obtain the proper time axis. We expect a 
    well-reasoned answer based on the theories discussed in class.
  </h3>
<p></p>
</li>
<li>
    <h3>Sum the 8 action potential trains in order to obtain the EMG signal. Plot the EMG signal as 
    function of time (in the time interval 10-10.5 s).
    </h3>
    <p></p>
</li>
</ol>
