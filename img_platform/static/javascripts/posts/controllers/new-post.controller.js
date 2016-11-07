//inject angular file upload directives and services.
(function (){
  'use strict';

  angular
    .module('platform.posts.controllers')
    .controller('NewPostController', NewPostController);

    NewPostController.$inject = ['$scope', '$http'];
    function NewPostController ($scope, $http) {
      console.log('in the controller');
      $scope.uploadFile = function(files) {
        console.log(files);
        var fd = new FormData();
        var user_caption = $scope.user_caption;
        console.log(user_caption);
        //Take the first selected file
        fd.append("image", files[0]);
        fd.append("user_caption", user_caption);
        for (var pair of fd.entries()) {
            console.log(pair[0]+ ', ' + pair[1]);
        }

        $http.post('/api/v1/posts/', fd, {
          withCredentials: true,
          headers: {'Content-Type': undefined },
          transformRequest: angular.identity
        });

    };
  }
})();
