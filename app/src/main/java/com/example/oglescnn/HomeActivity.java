package com.example.oglescnn;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;


public class HomeActivity extends Activity {


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
    }

    public void enterCifar10(View view) {
        startActivity(new Intent(this, CifarActivity.class));
    }

    public void enterSqueezeNet(View view) {
        startActivity(new Intent(this, SqueezeNetActivity.class));
    }
}
