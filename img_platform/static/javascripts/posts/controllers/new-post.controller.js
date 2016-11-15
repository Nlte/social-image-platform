(function () {
  'use strict';

  angular
    .module('platform.posts.controllers')
    .controller('NewPostController', NewPostController);

  NewPostController.$inject = ['$rootScope', '$scope', 'Authentication', 'Snackbar', 'Posts'];


  function NewPostController($rootScope, $scope, Authentication, Snackbar, Posts) {
    var vm = this;

    vm.submit = submit;
    var fd = new FormData();
    $scope.uploadFile = function(files) {
      fd.append("image", files[0]);
      fd.append("title", vm.title);
    }


    function submit() {

      $scope.closeThisDialog();

      Posts.create(fd).then(createPostSuccessFn, createPostErrorFn);

      function createPostSuccessFn(data, status, headers, config) {
        Snackbar.show('Success! Post created.');
      }

      function createPostErrorFn(data, status, headers, config) {
        $rootScope.$broadcast('post.created.error');
        Snackbar.error(data.error);
      }
    }
  }
})();
