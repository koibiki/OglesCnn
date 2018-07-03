package com.example.oglescnn;

import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.View;

import com.example.cnnlib.utils.FileUtils;

import static android.content.pm.PackageManager.PERMISSION_GRANTED;

public class HomeActivity extends Activity {

    private String TAG = "HomeActivity";
    private String sRoot = Environment.getExternalStorageDirectory().getAbsolutePath();


    private ProgressDialog progressDialog;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        requestAllPower();
    }

    public void requestAllPower() {
        if (checkSelfPermission(
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PERMISSION_GRANTED) {

            requestPermissions(
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.READ_EXTERNAL_STORAGE}, 1);

        } else {
            initEnv();
        }
    }

    public void enterCNN(View view) {
        startActivity(new Intent(this, MainActivity.class));
    }

    private void initEnv() {
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage("正在复制模型文件，请稍等......");
        progressDialog.setCancelable(false);
        progressDialog.show();

        new Thread(() -> {
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/Cifar10_def_test.txt");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/Cifar10_def.txt");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/labels.txt");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/mean.msg");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/model_param_conv1.msg");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/model_param_conv2.msg");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/model_param_conv3.msg");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/model_param_ip1.msg");
            FileUtils.copyDBToSD(this, sRoot + "/Cifar10es", "net/model_param_ip2.msg");
            Log.d(TAG, "copy completed");
            progressDialog.dismiss();
        }).start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == 1) {
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] == PERMISSION_GRANTED) {
                    initEnv();
                } else {
                    finish();
                }
            }
        }
    }
}
