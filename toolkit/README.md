# Proto-CLIP Toolkit

This README explains each of the components of the Proto-CLIP toolkit and provides details on how to run each of them. The Proto-CLIP toolkit can be accessed by running `pip install proto_clip_toolkit`. However, for running the real world robot demo, you would need to clone the repository and follow the set of instructions given below. **Due to conflicts in the naming scheme inside proto-clip and hugging face, we recommend you create a separae conda environment if you want to use the proto clip toolkit**

The directory structure shown below describes each of the individual components of the Proto-CLIP toolkit:

```
./proto_clip_toolkit
    |--ros
        |--utils/
        |--scripts/
        |--proto_clip_node.py
        |--proto_clip_results_node.py
    |--utils
        |--tsne.py
        |--model_utils.py
    |--pos
        |--configs/
        |--verb_and_noun_tagger.py
    |--asr
        |--configs/
        |--asr_utils.py
        |--transcribe.py
        |--transcribe_with_pos.p
```

## Proto-CLIP real world demo

The real world demo described in the paper is a culmination of multiple individual systems that need to initialized separately. Since we require multiple systems to be run simultaneously, we recommend using the [Terminator](https://github.com/gnome-terminator/terminator) terminal on Ubuntu. The diagram below describes the system in detail. 

![The Block Diagram representation of the entire system. The numbers represent the order in which each node should be executed](./media/demo_block_diagram.jpg)


The details on running each of the nodes is given below:

1. The first step of the demo would be to start the fetch robot :). If you do not have a fetch robot available, you can setup the Fetch robot in Gazebo and use the same topic names as the original robot. The instructions for setting up gazebo are provided in the SceneReplica repository linked in Step 4.

2. The second step of the demo is to run the segmentation node. First, you would need to clone the following repository [UnseenObjectsWithMeanShift](https://github.com/YoungSean/UnseenObjectsWithMeanShift). Next, cd into the repository and run the following command on your terminal:

    ```
    ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh $GPU_ID
    ```
    
3. The third step of the demo is to run the Proto-CLIP node. You need to navigate into the `toolkit/ros` directory and run the following command.:

    ```
    ./scripts/run_proto_clip_node.sh
    ```

In case you want to try out different embeddings or a different pretrained-adapter, please modify their values in the config in the script file.

4. The final step of the demo is to run the grasping code. Clone the following repository [SceneReplica](https://github.com/IRVLUTD/SceneReplica). Then follow the instructions given below:
    - Replace `/seg_label_refined` in this [line](https://github.com/IRVLUTD/SceneReplica/blob/main/src/utils/image_listener.py#L154) to `/selected_seg_label`.
    - Replace `/seg_score` in this [line](https://github.com/IRVLUTD/SceneReplica/blob/main/src/utils/image_listener.py#L155) to `selected_seg_score`.
    - Replace the `slop_seconds` in this [line](https://github.com/IRVLUTD/SceneReplica/blob/main/src/utils/image_listener.py#L190) to 50.0 . In case, the next steps do not work for you please come back here and increase your slop seconds further.
    - Finally, you can now run the code. Follow the instructions in the README to setup model free grasping. The particular configuration we used in our demo is listed below:

        ```
            --grasp_method contact_gnet --seg_method msmformer --obj_order nearest_first --scene_idx 25
        ```

    The values supplied to the `obj_order` and the `scene_idx` arguments does not matter since the Proto-CLIP will supply only a single object to the grasping node.
    - The grasping code will prompt you to execute the actions, press enter to proceed to execute them. Once the grasping for an object is complete, you would need to close the code and run it again. Please ensure that you speak the next object for the Proto-CLIP node in 4 only after this code starts running again. This will be certainly painful and we are working on addressing this issue.

5. If you want to view the results in RViz similar to our paper, run the following command inside the `ros` directory as in step 4. 
    ```
    ./scripts/run_proto_clip_rviz_results_pub.sh
    ```

You can find the terminator window below for reference. The numbers represent the sequence number mentioned before.

![](./media/terminator_image.png)

## Automatic Speech Recognition(ASR)

The ASR module can be found inside the `asr` folder. There are two major functions exported by this module `transcribe` and `transcribe_with_pos`. The `transcribe` function takes in speech input and prints the output to the console. The `transcribe_with_pos` function is a modified function written for Proto-CLIP grasping. The function transcribes the speech and matches the action(verb) and object(noun) using the Part of Speech (POS) tagging module. The specifics of this module is described in the next section. However, when the module finds a correct action and object, the module stops and returns them. 

The ASR requires the user to specify the config which can be found in the `asr/configs` directory inside the toolkit. The parameters of the config are explained below:

```
{
    "model": "The name of the ASR model you want to use",
    "non_english": "Boolean specific to whisper which specifies if you want to transcribe to a non-english language",
    "energy_threshold": "The energy threshold controls the sensitivity of your microphone for it to start the transcription. Recommended Value is 1000",
    "record_timeout": "Timeout in seconds for recording",
    "phrase_timeout": "Timeout in seconds for the length of the phrase to record" ,
    "default_microphone": "Name of your microphone"
}
```

## Part of Speech Tagging(POS)

The Part of Speech(POS) tagging module currently uses flair underneath to perform tagging. We have included the `VerbAndNounTagger` class in the POS package. The module needs to be initialized with a list of acceptable verbs and nouns. The `tag_sentence` function finds the verb and the noun in the sentence that are present in the dictionary, if not it returns `None` for either of the values.

The ASR and POS module are provided in this toolkit for you to experiment with different models for ASR and POS based on your needs.

## Utils

We provide two utils for the proto clip toolkit. The first util is to plot the tSNE plot for text and image embeddings similar to the one shown in the Proto-CLIP paper. There are two functions present to plot the Proto-CLIP tSNE before and after training. **Before running tSNE please rename the datasets folder inside `proto-clip` to `proto_datasets`.

The second util is to load the pretrained visual and textual embeddings using the paths for the memory bank.
