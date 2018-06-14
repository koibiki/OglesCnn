package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initReluPro;

public class NonLinearLayer extends Layer {

    private NonLinearType mType;
    private int mNumGroupsY;
    private int mShaderPro;

    public enum NonLinearType {
        RELU, SIGMOID
    }


    public NonLinearLayer(Context context, LayerParams layerParams, NonLinearType type) {
        super(context, layerParams);
        this.mType = type;

        initNonlinear();
    }

    private void initNonlinear() {
        String csPath;
        if (mType == NonLinearType.RELU) {
            csPath = "relu.comp";
        } else {
            csPath = "sigmoid.comp";
        }
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initReluPro(mContext, csPath, mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutputTexID = ComputeRender.createTexture(mAttachID);
    }

    @Override
    public int forwardProc(int inputTexID) {
        ComputeRender.performNonLinear(mShaderPro, inputTexID, mOutputTexID, mNumGroupsY);
        return mOutputTexID;
    }
}
