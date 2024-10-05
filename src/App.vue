<script setup lang="ts">
// This starter template is using Vue 3 <script setup> SFCs
// Check out https://vuejs.org/api/sfc-script-setup.html#script-setup
// import Greet from "./components/Greet.vue";
import TwoMatrix from "./components/TwoMatrix.vue"
</script>

<template>
  <div class="h-full w-full flex flex-row">
    <div class="bg-gray-700 w-1/5 h-full p-3">
      <div class="m-auto shadow-2xl text-center p-3">
        Простые <br>операции
      </div>
      <div class="m-auto shadow-2xl text-center mt-4 p-3">
        Линейные <br>уравнения
      </div>
    </div>
    <div class="w-4/5 h-full p-3">
      <TwoMatrix />
    </div>
  </div>
</template>
