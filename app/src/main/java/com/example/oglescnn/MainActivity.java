package com.example.oglescnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.activate.ActivationFunc;
import com.example.cnnlib.layer.ConvolutionLayer;
import com.example.cnnlib.layer.InputLayer;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.MathUtils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        CnnNetwork cnnNetwork = new CnnNetwork();
        LayerParams inLayerParams = new LayerParams(32, 32, 4, 32, 32, 4);

        cnnNetwork.addLayer(new InputLayer(this, inLayerParams));

        LayerParams outLayerParams = new LayerParams(32, 32, 4, 32, 32, 4);

        List<Kennel> kennels = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            kennels.add(createKennel(i + 1));
        }

        cnnNetwork.addLayer(new ConvolutionLayer(this, outLayerParams, kennels, 1, Constants.S_PADDING_SAME, new int[]{1, 1}));
        cnnNetwork.run();
    }

    private Kennel createKennel(int num) {
        int width = 3;
        int height = 3;
        int channel = 4;
        float[] data = new float[width * height * channel];
        for (int i = 0; i < width * height * channel; i++) {
            data[i] = num;
        }
        return new Kennel(width, height, channel, data);
    }

}
